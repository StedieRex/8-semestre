package com.bubberspot.receiver;

import static com.sun.faces.facelets.util.Path.context;
import jakarta.annotation.Resource;
import jakarta.jms.ConnectionFactory;
import jakarta.jms.JMSConsumer;
import jakarta.jms.JMSContext;
import jakarta.jms.Queue;

public class MessangerReceiver {
    @Resource(mappedName="jms/MyConnectionFactory")
    private static ConnectionFactory connectionFactory;
    
    @Resource(mappedName="jms/MyQueue")
    private static Queue queue;
    
    public static void main(String[] args){
        JMSContext jmsContext = connectionFactory.createContext();
        JMSConsumer jmsConsumer = jmsContext.createConsumer(queue);
        System.out.println("reciving message -");
        String message = jmsConsumer.receiveBody(String.class);
        System.out.println("message recive - "+message);
       
    }
}
