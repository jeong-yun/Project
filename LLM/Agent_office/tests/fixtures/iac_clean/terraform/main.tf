# 오탐 검사용 '괜찮은' IaC — 내장 규칙이 아무것도 찾지 않아야 한다 (tests/test_infra_security.py)
terraform {
  required_version = ">= 1.6"
}

locals {
  office_cidr = "198.51.100.0/24"
  enabled     = var.env == "prod" ? true : false
}

variable "env" {
  type    = string
  default = "prod"
}

/* 예전 방식(인라인 블록)으로 암호화·로그를 건 버킷 */
resource "aws_s3_bucket" "reports" {
  bucket = "clean-reports"
  acl    = "private"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  logging {
    target_bucket = "clean-log-archive"
    target_prefix = "reports/"
  }
}

resource "aws_s3_bucket" "log_archive" {
  bucket = "clean-log-archive"
}

resource "aws_s3_bucket_acl" "log_archive" {
  bucket = aws_s3_bucket.log_archive.id
  acl    = "log-delivery-write"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "log_archive" {
  bucket = "clean-log-archive"
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_logging" "log_archive" {
  bucket        = aws_s3_bucket.log_archive.bucket
  target_bucket = aws_s3_bucket.log_archive.id
  target_prefix = "self/"
}

resource "aws_security_group" "web" {
  name = "clean-web"

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.office_cidr, "203.0.113.10/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_vpc_security_group_ingress_rule" "rdp_office" {
  security_group_id = aws_security_group.web.id
  cidr_ipv4         = "198.51.100.0/24"
  from_port         = 3389
  to_port           = 3389
  ip_protocol       = "tcp"
}

resource "aws_security_group_rule" "http_world" {
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.web.id
}

resource "aws_iam_policy" "deploy" {
  name   = "clean-deploy"
  policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["ecs:UpdateService", "ecs:DescribeServices"], "Resource": "*"},
    {"Effect": "Deny", "Action": "*", "Resource": "*", "Condition": {"Bool": {"aws:MultiFactorAuthPresent": "false"}}}
  ]
}
POLICY
}

data "aws_iam_policy_document" "read_reports" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.reports.arn}/*"]
  }
}

resource "aws_db_instance" "main" {
  identifier          = "clean-main"
  engine              = "postgres"
  instance_class      = "db.t3.micro"
  allocated_storage   = 20
  storage_encrypted   = true
  publicly_accessible = false
  tags                = { for k, v in var.extra_tags : k => v if v != "" }
}

variable "extra_tags" {
  type    = map(string)
  default = {}
}
