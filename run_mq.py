from config.backend import RabbitMQConfig
from task.task import TaskConsumer

if __name__ == "__main__":
    config = RabbitMQConfig()
    consumer = TaskConsumer(config)
    consumer.start_consuming()
