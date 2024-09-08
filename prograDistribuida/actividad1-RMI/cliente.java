import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class cliente {
    public static void main(String[] args) {
        try {
            int a = 5;
            int b = 3;

            // Conectar al Servidor de Suma
            Registry registrySuma = LocateRegistry.getRegistry("localhost", 1099);
            interfazPrincipal suma = (interfazPrincipal) registrySuma.lookup("ServidorSuma");
            System.out.println(suma.obtenerMensaje() + suma.operar(a, b));

            // Conectar al Servidor de Resta
            Registry registryResta = LocateRegistry.getRegistry("localhost", 1100);
            interfazPrincipal resta = (interfazPrincipal) registryResta.lookup("servidorResta");
            System.out.println(resta.obtenerMensaje() + resta.operar(a, b));

            // Conectar al Servidor de Multiplicación
            Registry registryMultiplicacion = LocateRegistry.getRegistry("localhost", 1101);
            interfazPrincipal multiplicacion = (interfazPrincipal) registryMultiplicacion.lookup("servidorMultiplicacion");
            System.out.println(multiplicacion.obtenerMensaje() + multiplicacion.operar(a, b));

        } catch (Exception e) {
            System.err.println("Excepción en cliente: " + e.toString());
            e.printStackTrace();
        }
    }
}

