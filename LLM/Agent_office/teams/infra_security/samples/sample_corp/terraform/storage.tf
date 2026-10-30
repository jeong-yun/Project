# 샘플상사(가상 회사) 클라우드 — 인프라 보안 팀 시험용 IaC. 실제 회사·계정·버킷이 아니다.

resource "aws_s3_bucket" "public_assets" {
  bucket = "samplecorp-public-assets"
  acl    = "public-read"

  tags = {
    Owner = "platform"
    Env   = "prod"
  }
}

resource "aws_s3_bucket" "audit_logs" {
  bucket = "samplecorp-audit-logs"

  tags = {
    Owner = "security"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "audit_logs" {
  bucket        = aws_s3_bucket.audit_logs.id
  target_bucket = aws_s3_bucket.audit_logs.id
  target_prefix = "access-log/"
}

resource "aws_s3_bucket_public_access_block" "audit_logs" {
  bucket                  = aws_s3_bucket.audit_logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
