import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class Cliente {
    public static void main(String[] args) {
        try {
            // Conectar al registro RMI en el localhost
            Registry registry = LocateRegistry.getRegistry("localhost", 1099);

            // Buscar el objeto remoto por su nombre
            Calculadora calculadora = (Calculadora) registry.lookup("Calculadora");

            // Invocar el método remoto
            int resultado = calculadora.sumar(5, 3);

            System.out.println("El resultado de la suma es: " + resultado);
        } catch (Exception e) {
            System.err.println("Excepción en el cliente: " + e.toString());
            e.printStackTrace();
        }
    }
}

