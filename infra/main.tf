# rxcite on AWS: API on ECS Fargate behind an ALB, Postgres + pgvector on RDS.
#
# Sized for a short demo on a small budget (~$0.10/hour while running):
#   - default VPC and public subnets, so no NAT gateway (~$1/day) is needed
#   - RDS is NOT publicly accessible; only the API's security group can reach it
#   - the database URL (with password) lives encrypted in SSM Parameter Store
#   - the task role can call exactly one Bedrock model, plus read the seed file
# `terraform destroy` removes everything, including the database (no final snapshot).

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 6.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "rxcite", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  name               = "rxcite"
  profile_arn        = "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.me.account_id}:inference-profile/${var.model_id}"
  foundation_model   = replace(var.model_id, "/^(us|eu|apac|global)\\./", "")
  seed_key           = "seed/chunks.jsonl.gz"
  image              = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
  database_url_param = "/${local.name}/database_url"
}

# ---------------------------------------------------------------- container registry

resource "aws_ecr_repository" "api" {
  name                 = local.name
  force_delete         = true # demo: allow destroy even with images inside
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---------------------------------------------------------------- security groups

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Public HTTP to the load balancer"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "api" {
  name        = "${local.name}-api"
  description = "API tasks: only the load balancer may connect"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress { # Bedrock, ECR, S3, RDS
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "Postgres: only the API tasks may connect"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}

# ---------------------------------------------------------------- database

resource "random_password" "db" {
  length  = 32
  special = false # keeps the connection URL simple to embed
}

resource "aws_db_subnet_group" "db" {
  name       = local.name
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_db_instance" "db" {
  identifier              = local.name
  engine                  = "postgres"
  engine_version          = "17"
  instance_class          = "db.t4g.micro"
  allocated_storage       = 20
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = local.name
  username                = local.name
  password                = random_password.db.result
  db_subnet_group_name    = aws_db_subnet_group.db.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  publicly_accessible     = false
  backup_retention_period = 0    # demo: no automated backups
  skip_final_snapshot     = true # demo: destroy cleanly
  deletion_protection     = false
}

resource "aws_ssm_parameter" "database_url" {
  name  = local.database_url_param
  type  = "SecureString"
  value = "postgresql://${local.name}:${random_password.db.result}@${aws_db_instance.db.address}:5432/${local.name}?sslmode=require"
}

# ---------------------------------------------------------------- seed data

resource "aws_s3_bucket" "seed" {
  bucket_prefix = "${local.name}-seed-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "seed" {
  bucket                  = aws_s3_bucket.seed.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "seed" {
  bucket = aws_s3_bucket.seed.id
  key    = local.seed_key
  source = "${path.module}/../data/seed/chunks.jsonl.gz"
  etag   = filemd5("${path.module}/../data/seed/chunks.jsonl.gz")
}

# ---------------------------------------------------------------- IAM

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: used by ECS itself to pull the image, write logs, read the secret.
resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_ssm" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameters"]
      Resource = [aws_ssm_parameter.database_url.arn]
    }]
  })
}

# Task role: what the application code may do. Least privilege.
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy" "task" {
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeViaInferenceProfile"
        Effect   = "Allow"
        Action   = "bedrock:InvokeModel"
        Resource = local.profile_arn
      },
      {
        Sid       = "InvokeUnderlyingModelOnlyThroughProfile"
        Effect    = "Allow"
        Action    = "bedrock:InvokeModel"
        Resource  = "arn:aws:bedrock:*::foundation-model/${local.foundation_model}"
        Condition = { StringEquals = { "bedrock:InferenceProfileArn" = local.profile_arn } }
      },
      {
        Sid      = "ReadSeedFile"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.seed.arn}/${local.seed_key}"
      },
    ]
  })
}

# ---------------------------------------------------------------- load balancer

resource "aws_lb" "api" {
  name               = local.name
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "api" {
  name        = local.name
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = data.aws_vpc.default.id
  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 5
    interval            = 15
  }
  deregistration_delay = 10
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ---------------------------------------------------------------- ECS

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}"
  retention_in_days = 3
}

resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_ecs_task_definition" "api" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64" # image is built on Apple Silicon; Graviton is cheaper too
  }
  container_definitions = jsonencode([{
    name         = "api"
    image        = local.image
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "RXCITE_PROVIDER", value = "bedrock" },
      { name = "RXCITE_MODEL", value = var.model_id },
      { name = "RXCITE_REGION", value = var.region },
      { name = "RXCITE_MODE", value = "vector" },
      { name = "RXCITE_REFUSE_BELOW", value = "0.74" },
    ]
    secrets = [{ name = "RXCITE_DATABASE_URL", valueFrom = aws_ssm_parameter.database_url.arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.api.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name                              = local.name
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = 1
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 120 # models load on first start
  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true # public subnets without a NAT gateway
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.http]
}
