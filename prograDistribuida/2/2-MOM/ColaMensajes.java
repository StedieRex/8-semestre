import java.util.LinkedList;
import java.util.Queue;
/**
 * ColaMensajes es una clase que implementa la interfaz MessageQueue.
 * Proporciona una cola de mensajes segura para hilos utilizando una LinkedList para almacenar mensajes.
 * 
 * La clase incluye los siguientes métodos:
 * 
 * - put(String message): Agrega un mensaje a la cola y notifica a los oyentes.
 * - get(): Recupera y elimina la cabeza de la cola, devolviendo null si la cola está vacía.
 * - pull(): Recupera pero no elimina la cabeza de la cola, devolviendo null si la cola está vacía.
 * - notifelisteners(): Notifica a los oyentes sobre nuevos mensajes.
 * 
 * La cola está marcada como final para prevenir modificaciones.
 */
class  ColaMensajes implements MessageQueue {
    //el final es para que no se pueda modificar
    final private Queue<String> queue = new LinkedList<>();

    @Override
    public void put(String message) {
        queue.add(message);
        System.out.println("Mensaje agregado: " + message);
        notifelisteners();
    }

    @Override
    public synchronized  String get() {
        if (queue.isEmpty()) {
            return null;
        }
        String messege = queue.poll();
        System.out.println("Mensaje obtenido: " + messege);
        return messege;
    }

    @Override
    public synchronized  String pull() {
        if (queue.isEmpty()) {
            return null;
        }
        String messege = queue.peek();
        System.out.println("Mensaje extraido: " + messege);
        return messege;
    }

    @Override
    public void notifelisteners() {
        System.out.println("Notificando a los usuarios de nuevos mensajes");
    }
}