import java.util.LinkedList;
import java.util.Queue;
import javax.annotation.processing.Messager;
interface  coso {
    void put(String message);
    String get();
    String pull();
    void notifelisteners();
}

class colaMensajes implements Messager.Queue{
    private Queve<String>queve=new linkedlist<>();
    @Override
    public synchronized void put(String message){
        queve.add(message);
        System.out.print("Mensaje agregado: "+message);
        notifylisteners();
    }
}


@overraide