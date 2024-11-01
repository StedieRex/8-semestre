
package com.hubberspot.sender;

import jakarta.annotation.Resource;
import jakarta.jms.ConnectionFactory;
import jakarta.jms.JMSContext;
import jakarta.jms.JMSProducer;
import jakarta.jms.Queue;

public class MessageSender {
   @Resource(mappedName="jms/MyConnectionFactory")
    private static ConnectionFactory connectionFactory; 
   
   @Resource(mappedName = "jms/MyQueue")
   private static Queue queue;
   
   public static void main(String[] args){
       JMSContext jmsContext = connectionFactory.createContext();
       JMSProducer jmsProducer = jmsContext.createProducer();
       
       String message = "hellos JMS";
       System.out.println("sending message to jms -");
       jmsProducer.send(queue, message);
       System.out.println("Message send successfully");
   }
}
