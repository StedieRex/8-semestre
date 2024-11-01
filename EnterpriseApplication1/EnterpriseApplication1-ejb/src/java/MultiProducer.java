
import jakarta.annotation.Resource;
import jakarta.jms.ConnectionFactory;
import jakarta.jms.Destination;
import jakarta.jms.JMSContext;
import jakarta.jms.JMSProducer;
import jakarta.jms.Queue;



public class MultiProducer {
    @Resource(lookup = "java:comp/DefaultJMSConnectionFactory")
    private static ConnectionFactory connectionFactory;

    @Resource(lookup = "jms/MyQueue")
    private static Queue queue;

    public static void main(String[] args) {
        try (JMSContext context = connectionFactory.createContext()) {
            JMSProducer producer = context.createProducer();
            for (int i = 0; i < 10; i++) {
                String message = "Mensaje " + i;
                int priority = (int) (Math.random() * 10);
                producer.setPriority(priority).send((Destination) queue, message);
                System.out.println("Enviado: " + message + " con prioridad " + priority);
                Thread.sleep(1000); // Simular intervalo entre envíos
            }
        } catch (Exception e) {
            System.err.println("Error en el productor: " + e.getMessage());
        }
    }
}
