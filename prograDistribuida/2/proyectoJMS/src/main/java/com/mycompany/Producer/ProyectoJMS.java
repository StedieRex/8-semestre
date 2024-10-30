package com.mycompany.Producer;

import javax.jms.*;
import org.apache.activemq.ActiveMQConnectionFactory;

public class ProyectoJMS {

    public static void main(String[] args) {
        try {
// Crear conexión
            ConnectionFactory connectionFactory = new ActiveMQConnectionFactory("tcp://localhost:61616");
            Connection connection = connectionFactory.createConnection();
            connection.start();
// Crear sesión
            Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Destination destination = session.createQueue("TestQueue");
// Crear productor
            MessageProducer producer = session.createProducer(destination);
            TextMessage message = session.createTextMessage("Hola desde el productor!");
// Enviar mensaje
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
