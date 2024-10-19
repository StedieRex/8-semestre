import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class Servidor {
    public static void main(String[] args) {
        try {
            // args[0] debe ser la IP de la máquina que actúa como servidor en la red local (por ejemplo, "192.168.1.100")
            System.setProperty("java.rmi.server.hostname", args[0]);

            // Crea el registro de RMI en el puerto 2320
            Registry registro = LocateRegistry.createRegistry(2320);

            // Enlaza el objeto remoto al nombre "SistemasDistribuidos"
            registro.rebind("SistemasDistribuidos", new ObjetoRemoto());

            System.out.println("Servidor listo en " + args[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
