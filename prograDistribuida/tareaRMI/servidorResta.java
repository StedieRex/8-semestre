import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class servidorResta extends UnicastRemoteObject implements interfazPrincipal {

    protected servidorResta() throws RemoteException {
        super();
    }

    @Override
    public int operar(int a, int b) throws RemoteException {
        return a - b;
    }

    @Override
    public String obtenerMensaje() throws RemoteException {
        return "La resta de los números es: ";
    }

    public static void main(String[] args) {
        try {
            servidorResta resta = new servidorResta();
            Registry registry = LocateRegistry.createRegistry(1100);
            registry.rebind("servidorResta", resta);
            System.out.println("Saludos desde el servidro que resta.");
        } catch (Exception e) {
            System.err.println("Excepción en Servidor Resta: " + e.toString());
            e.printStackTrace();
        }
    }
}
