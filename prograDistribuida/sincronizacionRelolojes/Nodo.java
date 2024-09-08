import java.util.Random;

public class Nodo {
    private int id;
    private int reloj;

    public Nodo(int id) {
        this.id = id;
        // inicializar el reloj con un valor aleatorio
        this.reloj = new Random().nextInt(100);
    }

    public int getReloj() {
        return reloj;
    }

    public void setReloj(int nuevoReloj) {
        this.reloj = nuevoReloj;
    }

    public int getId() {
        return id;
    }

    public void imprimiendoReloj() {
        System.out.println("Nodo " + id + "tiene el reloj en:" + reloj);
    }

}
