package com.example;

import javax.jms.Connection;
import javax.jms.JMSException;
import javax.jms.MessageConsumer;
import javax.jms.Queue;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.apache.activemq.ActiveMQConnectionFactory;

public class Consumer {

    public static void main(String[] args) {
        Connection connection = null;
        Session session = null;
        MessageConsumer consumer = null;
        
        try {
            // Crear conexión con ActiveMQ
            ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("tcp://localhost:61616");
            connection = connectionFactory.createConnection();
            connection.start();

            // Crear sesión
            session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Queue queue = session.createQueue("testQueue");

            // Crear consumidor para recibir mensajes
            consumer = session.createConsumer(queue);
            consumer.setMessageListener(message -> {
                if (message instanceof TextMessage) {
                    try {
                        System.out.println("Mensaje recibido: " + ((TextMessage) message).getText());
                    } catch (JMSException e) {
                        e.printStackTrace();
                    }
                } else {
                    System.out.println("Mensaje no es del tipo TextMessage");
                }
            });

            System.out.println("Esperando mensajes...");

            // Para un entorno de producción, este hilo debe mantenerse activo indefinidamente.
            // Aquí lo dejamos en pausa por 10 segundos como prueba.
            Thread.sleep(10000);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Limpiar recursos de forma adecuada en el bloque finally
            try {
                if (consumer != null) consumer.close();
                if (session != null) session.close();
                if (connection != null) connection.close();
            } catch (JMSException e) {
                e.printStackTrace();
            }
        }
    }
}
