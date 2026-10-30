# 샘플상사(가상 회사) IAM — 시험용.

resource "aws_iam_policy" "ops_admin" {
  name        = "samplecorp-ops-admin"
  description = "Temporary policy for the ops team"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "*"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "app_read_logs" {
  name = "samplecorp-app-read-logs"
  role = "samplecorp-app-role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = ["arn:aws:s3:::samplecorp-audit-logs", "arn:aws:s3:::samplecorp-audit-logs/*"]
      }
    ]
  })
}
