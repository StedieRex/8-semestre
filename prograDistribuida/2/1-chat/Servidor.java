import java.rmi.registry.*;
public class Servidor {
    public static void main(String []args){
        try {
            System.setProperty("java.rmi.server.hostname",args[0]);
            Registry registro = LocateRegistry.createRegistry(2320);// se crea el registro en el puerto 2320 para que los clientes se conecten
            registro.rebind("SistemasDistribuidos", new ObjetoRemoto());
        } catch (Exception e) {e.printStackTrace();}
    }
}
