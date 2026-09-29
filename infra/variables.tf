variable "region" {
  description = "AWS region (Bedrock model access must be enabled here)."
  type        = string
  default     = "us-east-1"
}

variable "model_id" {
  description = "Bedrock inference profile ID for the answering model."
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "image_tag" {
  description = "Tag of the API image pushed to ECR."
  type        = string
  default     = "latest"
}

variable "allowed_cidr" {
  description = "Who may reach the public load balancer (narrow this to your IP for demos)."
  type        = string
  default     = "0.0.0.0/0"
}
