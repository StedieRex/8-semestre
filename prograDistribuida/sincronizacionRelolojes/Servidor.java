public class Servidor {
    private Nodo[] nodos;

    public Servidor(Nodo[] nodos){
        this.nodos = nodos;
    }

    public void SincronizacionRelojes(){
        // calcular la media de los relojes
        int sumaRelojes = 0;
        for(Nodo nodo: nodos){
            sumaRelojes += nodo.getReloj();
        }
        int promedio = sumaRelojes / nodos.length;

        System.out.println("El servidor calcula el reloj promedio: " + promedio);

        // ajustar los relojes de los nodos promedio
        for(Nodo nodo: nodos){
            nodo.setReloj(promedio);
            System.out.println("El reloj del nodo "+nodo.getId()+" se ajusta a: "+promedio);
        }
    }
}
