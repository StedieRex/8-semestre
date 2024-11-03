package com.hubberspot.sender;

import javax.annotation.Resource;
import javax.jms.*;

public class MessageSender {
    @Resource(lookup = "java:comp/DefaultJMSConnectionFactory")
    private static ConnectionFactory connectionFactory;

    @Resource(lookup = "jms/myQueue")
    private static Queue queue;

    public static void main(String[] args) {
        try (JMSContext context = connectionFactory.createContext()) {
            JMSProducer producer = context.createProducer();
            for (int i = 0; i < 10; i++) {
                String message = "Mensaje " + i;
                int priority = (int) (Math.random() * 10);
                producer.setPriority(priority).send(queue, message);
                System.out.println("Enviado: " + message + " con prioridad " + priority);
                Thread.sleep(1000); // Simular intervalo entre envíos
            }
        } catch (Exception e) {
            System.err.println("Error en el productor: " + e.getMessage());
        }
    }
}
