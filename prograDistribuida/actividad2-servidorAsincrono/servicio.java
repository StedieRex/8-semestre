import java.rmi.Remote;
import java.rmi.RemoteException;

public interface servicio extends Remote {
    public String procesarDatos(String nombre) throws RemoteException;
}