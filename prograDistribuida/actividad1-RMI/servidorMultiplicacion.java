import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class servidorMultiplicacion extends UnicastRemoteObject implements interfazPrincipal {

    protected servidorMultiplicacion() throws RemoteException {
        super();
    }

    @Override
    public int operar(int a, int b) throws RemoteException {
        return a * b;
    }

    @Override
    public String obtenerMensaje() throws RemoteException {
        return "La multiplicación de los números es: ";
    }

    public static void main(String[] args) {
        try {
            servidorMultiplicacion multiplicacion = new servidorMultiplicacion();
            Registry registry = LocateRegistry.createRegistry(1101);
            registry.rebind("servidorMultiplicacion", multiplicacion);
            System.out.println("Saludo desde el servidor que multiplica.");
        } catch (Exception e) {
            System.err.println("Excepción en Servidor Multiplicación: " + e.toString());
            e.printStackTrace();
        }
    }
}

