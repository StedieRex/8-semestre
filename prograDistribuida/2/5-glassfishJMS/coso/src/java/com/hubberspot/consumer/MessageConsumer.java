/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.hubberspot.consumer;

import javax.annotation.Resource;
import javax.jms.*;

public class MessageConsumer implements MessageListener {
    @Resource(lookup = "java:comp/DefaultJMSConnectionFactory")
    private static ConnectionFactory connectionFactory;

    @Resource(lookup = "jms/myQueue")
    private static Queue queue;

    public static void main(String[] args) {
        try (JMSContext context = connectionFactory.createContext()) {
            JMSConsumer consumer = context.createConsumer(queue);
            consumer.setMessageListener(new MessageConsumer());
            System.out.println("Esperando mensajes...");
            Thread.sleep(20000); // Escuchar durante 20 segundos
        } catch (Exception e) {
            System.err.println("Error en el consumidor: " + e.getMessage());
        }
    }

    @Override
    public void onMessage(Message message) {
        try {
            if (message instanceof TextMessage) {
                TextMessage textMessage = (TextMessage) message;
                System.out.println("Mensaje recibido: " + textMessage.getText());
            }
        } catch (JMSException e) {
            System.err.println("Error al procesar el mensaje: " + e.getMessage());
        }
    }
}

