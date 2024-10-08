public class SincronizacionRelojes {
    public static void main(String[] args){
        // Creamos los nodos en el sistema
        Nodo nodo1 = new Nodo(1);
        Nodo nodo2 = new Nodo(2);
        Nodo nodo3 = new Nodo(3);
        Nodo nodo4 = new Nodo(4);

        Nodo[] nodos = {nodo1, nodo2, nodo3, nodo4};

        // Imprimimos los relojes inciales
        System.err.println("Relojes antes de la sincronizacion: ");
        for(Nodo nodo: nodos){
            nodo.imprimiendoReloj();
        }

        //creamos el servidor y sincronizamo los relojes
        Servidor servidor = new Servidor(nodos);
        servidor.SincronizacionRelojes();

        // Imprimimos los relojes despues de la sincronizacion
        System.out.println("\nRelojes despues de la sincronizacion: ");
        for (Nodo nodo : nodos) {
            nodo.imprimiendoReloj();
        }
    }
}
