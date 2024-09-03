import java.rmi.Remote;
import java.rmi.RemoteException;

public interface interfazPrincipal extends Remote {
    public int operar(int a, int b) throws RemoteException;
    public String obtenerMensaje() throws RemoteException;
}

