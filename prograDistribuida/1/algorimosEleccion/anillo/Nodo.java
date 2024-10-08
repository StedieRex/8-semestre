
public class Nodo {
    int id;
    Nodo siguiente;
    boolean esLider=false;

    public Nodo(int id){
        this.id=id;
    }

    public void enviarMensajeEleccion(int idMayor){
        System.out.println("Nodo"+this.id+"recibió el mensaje de ID mayor: conecta al primero "+idMayor);
        if(idMayor>this.id){
            siguiente.enviarMensajeEleccion(idMayor);
        }else if(this.id>idMayor){
            siguiente.enviarMensajeEleccion(this.id);
        }else{
            System.out.println("Nodo "+this.id+" ha sido elegido como líder.");
            esLider=true;
        }
    }
}
