/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author Luis Pach
 */
// Crear conexión
import javax.jms.*;
import org.apache.activemq.ActiveMQConnectionFactory;

public class Consumidor {

    public static void main(String[] args) {
        try {
            ConnectionFactory connectionFactory = new ActiveMQConnectionFactory("tcp://localhost:61616");
            Connection connection = connectionFactory.createConnection();
            connection.start();
// Crear sesión
            Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Destination destination = session.createQueue("TestQueue");
// Crear consumidor
            MessageConsumer consumer = session.createConsumer(destination);
// Recibir mensaje
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
// Mantener el programa en ejecución para recibir mensajes
            System.out.println("Esperando mensajes...");
            Thread.sleep(10000); // Espera 10 segundos para recibir mensajes
// Limpiar recursos
            consumer.close();
            session.close();
            connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
