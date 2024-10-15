import java.util.LinkedList;
import java.util.Queue;
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