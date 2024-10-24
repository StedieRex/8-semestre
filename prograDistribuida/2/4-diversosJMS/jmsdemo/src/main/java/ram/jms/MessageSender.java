package ram.jms;

import javax.jms.Connection;
import javax.jms.ConnectionFactory;
import javax.jms.Destination;
import javax.jms.JMSException;
import javax.jms.MessageProducer;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.apache.activemq.ActiveMQConnection;
import org.apache.activemq.ActiveMQConnectionFactory;

/*
 * This class is used to send a text message to the queue.
 */
public class MessageSender
{

	/*
	 * URL of the JMS server. DEFAULT_BROKER_URL will just mean that JMS server is on localhost
	 * 
	 * default broker URL is : tcp://localhost:61616"
	 */
	private static String url = ActiveMQConnection.DEFAULT_BROKER_URL;

	/*
	 * Queue Name.You can create any/many queue names as per your requirement.
	 */
	private static String queueName = "MESSAGE_QUEUE";

	public static void main(String[] args) throws JMSException
	{
		System.out.println("url = " + url);

		/*
		 * Getting JMS connection from the JMS server and starting it
		 */
		ConnectionFactory connectionFactory = new ActiveMQConnectionFactory(url);
		Connection connection = connectionFactory.createConnection();
		connection.start();

		/*
		 * Creating a non transactional session to send/receive JMS message.
		 */
		Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);

		/*
		 * The queue will be created automatically on the server.
		 */
		Destination destination = session.createQueue(queueName);

		/*
		 * Destination represents here our queue 'MESSAGE_QUEUE' on the JMS server.
		 * 
		 * MessageProducer is used for sending messages to the queue.
		 */
                for(int i=1;i<101;i++){
                    MessageProducer producer = session.createProducer(destination);
                    TextMessage message = session.createTextMessage("Mensaje no: "+ i);
                    producer.send(message);
                }

		connection.close();
	}
}
