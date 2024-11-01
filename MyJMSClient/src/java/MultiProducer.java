

import jakarta.jms.ConnectionFactory;
import jakarta.jms.JMSContext;
import jakarta.jms.JMSProducer;
import jakarta.jms.Queue;
import javax.naming.InitialContext;
import javax.naming.NamingException;

public class MultiProducer {

    public static void main(String[] args) {
        try {
            InitialContext ctx = new InitialContext();
            ConnectionFactory connectionFactory = 
                (ConnectionFactory) ctx.lookup("jms/MyConnectionFactory");
            Queue queue = (Queue) ctx.lookup("jms/MyQueue");

            try (JMSContext jmsContext = connectionFactory.createContext()) {
                JMSProducer jmsProducer = jmsContext.createProducer();
                String message = "Hello JMS!";
                System.out.println("Sending message: " + message);
                jmsProducer.send(queue, message);
                System.out.println("Message sent successfully");
            }
        } catch (NamingException e) {
            e.printStackTrace();
        }
    }
}
