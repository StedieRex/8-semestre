import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class CalculadoraImpl extends UnicastRemoteObject implements Calculadora {

    protected CalculadoraImpl() throws RemoteException {
        super();
    }

    @Override
    public int sumar(int a, int b) throws RemoteException {
        return a + b;
    }

    public static void main(String[] args) {
        try {
            // Crear una instancia del objeto remoto
            Calculadora calculadora = new CalculadoraImpl();

            // Registrar el objeto remoto con el registro RMI
            Registry registry = LocateRegistry.createRegistry(1099);
            registry.rebind("Calculadora", calculadora);

            System.out.println("Servidor de Calculadora listo.");
        } catch (Exception e) {
            System.err.println("Excepción en el servidor de Calculadora: " + e.toString());
            e.printStackTrace();
        }
    }
}
