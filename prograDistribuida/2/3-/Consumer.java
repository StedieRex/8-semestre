import javax.jms.*;
import org.apache.activemq.ActiveMQConnectionFactory;

public class Consumer {
    public static void main(String[] args) {
        try {
            // Crear conexión[^1^][1]
            ConnectionFactory connectionFactory = new ActiveMQConnectionFactory("tcp://localhost:61616");
            Connection connection = connectionFactory.createConnection();
            connection.start();

            // Crear sesión
            Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Destination destination = session.createQueue("TestQueue");

            // Crear consumidor
            MessageConsumer consumer = session.createConsumer(destination);
            consumer.setMessageListener(new MessageListener() {
                public void onMessage(Message message) {
                    if (message instanceof TextMessage) {
                        try {
                            System.out.println("Mensaje recibido: " + ((TextMessage) message).getText());
                        } catch (JMSException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });

            // Mantener el programa en ejecución para recibir mensajes[^24^][24]
            System.out.println("Esperando mensajes...");
            Thread.sleep(10000); // Espera 10 segundos para recibir mensajes[^25^][25]

            // Limpiar recursos
            consumer.close();
            session.close();
            connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
