import java.rmi.registry.*;

public class Servidor1 {
    public static void main(String[] args) {
        try {
            // Create a registry
            Registry registry = LocateRegistry.createRegistry(2320);
            // Bind the object to the registry
            registry.rebind("sistemaDistribuido1", new objetoRemoto1());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}