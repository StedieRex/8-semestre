import java.rmi.Naming;
import java.rmi.registry.LocateRegistry;
public class servidor {
    public static void main(String[] args) {
        try {
            servicioImpl servicio = new servicioImpl();
            LocateRegistry.createRegistry(1099);
            Naming.rebind("rmi://localhost:1099/servicio", servicio);
            System.out.println("Servidor iniciado...");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
