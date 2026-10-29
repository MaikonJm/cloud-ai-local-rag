
# Amazon SQS — Dead Letter Queue

Categoria: AWS
Serviços: Amazon SQS, CloudWatch
Infraestrutura como código: Terraform

O Amazon Simple Queue Service (SQS) é um serviço
gerenciado de filas de mensagens da AWS.

Uma Dead Letter Queue (DLQ) recebe mensagens
que não foram processadas com sucesso após
atingirem o limite de recebimentos configurado.

As filas Standard e FIFO podem utilizar DLQs,
desde que a DLQ seja do tipo compatível.

No Terraform, podemos criar filas utilizando
o recurso aws_sqs_queue.

A propriedade redrive_policy permite associar
uma fila principal à sua Dead Letter Queue.

O CloudWatch pode monitorar a quantidade de
mensagens presentes na DLQ e gerar alarmes.
