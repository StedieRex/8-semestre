import java.util.Random;
public class AlgoritmoDelAnillo {
    public static void main(String[] args){
        int numNodos=5;
        Nodo[] nodos=new Nodo[numNodos];

        Random random = new Random();
        for(int i=0;i<numNodos;i++){
            nodos[i]=new Nodo(random.nextInt(100));
            System.out.println("Nodo "+i+" tiene ID: "+nodos[i].id);
        }

        for(int i=0;i<numNodos;i++){
            nodos[i].siguiente=nodos[(i+1)%numNodos];
        }
        System.err.println("\nIniciando la elección desde el Nodo "+nodos[0].id);
        nodos[0].enviarMensajeEleccion(nodos[0].id);
    }
}
