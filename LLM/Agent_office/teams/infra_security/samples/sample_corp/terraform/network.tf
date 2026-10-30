# 샘플상사(가상 회사) 네트워크 — 시험용. 사내 대역은 문서 전용 IP(RFC 5737)로 적었다.

resource "aws_security_group" "bastion" {
  name        = "samplecorp-bastion"
  description = "Bastion host"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH from anywhere"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "bastion_rdp" {
  type              = "ingress"
  description       = "RDP for vendor support"
  from_port         = 3389
  to_port           = 3389
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.bastion.id
}

resource "aws_security_group" "internal_ssh" {
  name   = "samplecorp-internal-ssh"
  vpc_id = var.vpc_id

  ingress {
    description = "SSH from office network"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["198.51.100.0/24"]
  }
}

variable "vpc_id" {
  type = string
}
