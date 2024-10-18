public class Main {
    public static void main(String[] args) {
        ColaMensajes cola = new ColaMensajes();
        Thread productor = new Thread(()->
        {
            for (int i = 0; i < 5; i++) {
                cola.put("Mensaje " + (i+1));
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        Thread consumidor = new Thread(()->
        {
            for (int i = 0; i < 5; i++) {
                cola.get();
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        productor.start();
        consumidor.start();

        
    }
}
