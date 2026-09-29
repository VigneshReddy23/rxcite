output "api_url" {
  description = "Public URL of the API."
  value       = "http://${aws_lb.api.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "seed_s3_uri" {
  value = "s3://${aws_s3_bucket.seed.id}/${local.seed_key}"
}

output "cluster" {
  value = aws_ecs_cluster.main.name
}

output "seed_task_network" {
  description = "Subnets and security group for the one-off seed task."
  value = {
    subnets        = data.aws_subnets.default.ids
    security_group = aws_security_group.api.id
  }
}
