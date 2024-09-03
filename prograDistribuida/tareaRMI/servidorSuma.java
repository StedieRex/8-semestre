import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class servidorSuma extends UnicastRemoteObject implements interfazPrincipal {

    protected servidorSuma() throws RemoteException {
        super();
    }

    @Override
    public int operar(int a, int b) throws RemoteException {
        return a + b;
    }

    @Override
    public String obtenerMensaje() throws RemoteException {
        return "La suma de los números es: ";
    }

    public static void main(String[] args) {
        try {
            servidorSuma suma = new servidorSuma();
            Registry registry = LocateRegistry.createRegistry(1099);
            registry.rebind("servidorSuma", suma);
            System.out.println("Saludos desde el servidor de suma.");
        } catch (Exception e) {
            System.err.println("Excepción en Servidor Suma: " + e.toString());
            e.printStackTrace();
        }
    }
}

