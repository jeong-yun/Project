# 샘플상사(가상 회사) 주문 DB — 시험용.

resource "aws_db_instance" "orders" {
  identifier                  = "samplecorp-orders"
  engine                      = "postgres"
  engine_version              = "16.3"
  instance_class              = "db.t3.micro"
  allocated_storage           = 20
  username                    = "orders_admin"
  manage_master_user_password = true
  storage_encrypted           = false
  publicly_accessible         = false
  backup_retention_period     = 7
  skip_final_snapshot         = false
}
