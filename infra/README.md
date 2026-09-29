# Deploying rxcite to AWS

Creates: ECR repository, RDS Postgres 17 (`db.t4g.micro`, pgvector, not public),
ECS Fargate service (ARM64, 1 vCPU / 2 GB) behind an ALB, IAM roles, an SSM
SecureString for the database URL, and a private S3 bucket for the seed file.

Cost while running: roughly **$0.10/hour** (RDS + Fargate + ALB). No NAT gateway.
Tear it down with `terraform destroy` when you're done.

Prerequisites: AWS credentials (`aws login` or similar), Bedrock access to the
model in `variables.tf`, Docker, Terraform ≥ 1.9, and an indexed local database.

```bash
# 0. Export the indexed chunks + embeddings (from the local database)
rxcite export-chunks                       # -> data/seed/chunks.jsonl.gz

cd infra
terraform init

# 1. Registry first, then push the image (built on Apple Silicon = ARM64)
terraform apply -target=aws_ecr_repository.api
REPO=$(terraform output -raw ecr_repository_url)
docker build -t rxcite-api:latest ..
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${REPO%%/*}"
docker tag rxcite-api:latest "${REPO}:latest" && docker push "${REPO}:latest"

# 2. Everything else
terraform apply

# 3. Seed the database with a one-off task (reads the seed file from S3)
SEED=$(terraform output -raw seed_s3_uri)
NET=$(terraform output -json seed_task_network)
SUBNETS=$(echo "$NET" | python3 -c "import sys,json; print(','.join(json.load(sys.stdin)['subnets']))")
SG=$(echo "$NET" | python3 -c "import sys,json; print(json.load(sys.stdin)['security_group'])")
aws ecs run-task --cluster rxcite --task-definition rxcite --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"api\",\"command\":[\"rxcite\",\"import-chunks\",\"--from\",\"$SEED\"]}]}"

# 4. Use it
URL=$(terraform output -raw api_url)
curl -X POST "$URL/ask" -H 'content-type: application/json' -d '{"question": "Does gabapentin make you sleepy?"}'

# 5. Remove everything
terraform destroy
```

Note for zsh users: write `"${REPO}:latest"`, not `$REPO:latest`. zsh treats
`:l` after a variable as a "lowercase" modifier and mangles the name.
