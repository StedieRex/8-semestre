import org.apache.activemq.ActiveMQConnectionFactory;

import javax.jms.*;

public class Producer {

    public static void main(String[] args) {
        try {
            // Crear conexión con ActiveMQ
            ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("tcp://localhost:61616");
            Connection connection = connectionFactory.createConnection();
            connection.start();

            // Crear sesión
            Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Queue queue = session.createQueue("testQueue");

            // Crear productor y enviar un mensaje
            MessageProducer producer = session.createProducer(queue);
            TextMessage message = session.createTextMessage("Hola, este es un mensaje simple.");
            producer.send(message);
            System.out.println("Mensaje enviado: " + message.getText());

            // Limpiar recursos
            producer.close();
            session.close();
            connection.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
