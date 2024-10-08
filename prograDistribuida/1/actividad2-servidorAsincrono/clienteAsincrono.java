
public class clienteAsincrono {
    public static void main(String[] args) {
        try {
            servicio ser = (servicio) java.rmi.Naming.lookup("rmi://localhost:1099/servicio"); 
            new Thread(() -> {
                try {
                    System.out.println("iniciando llamada asincrónica...");
                    String resultado = ser.procesarDatos("Dato de prueba");
                    System.out.println("Resultado resivido asincronicamente: " + resultado);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}