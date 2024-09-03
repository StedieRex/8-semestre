import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class servicioImpl extends UnicastRemoteObject implements servicio {
    protected servicioImpl() throws RemoteException {
        super();
    }

    @Override
    public String procesarDatos(String dato) throws RemoteException{
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "Procesado: " + dato;
    }
    
}
