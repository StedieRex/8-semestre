import java.rmi.*;
import java.rmi.server.*;

public class ObjetoRemoto extends UnicastRemoteObject implements IObjetoRemoto{
    private String mensaje1, mensaje2, alias1, alias2;
    
    public ObjetoRemoto() throws RemoteException {
        super();  
    }

    public void EnviaMensaje(String mensaje, int idpc) throws RemoteException {
        if(idpc == 1)
            mensaje1 = mensaje;
        if(idpc == 2)
            mensaje2 = mensaje;
        System.out.println(mensaje);
    }

    public String RecibeMensajePc1() throws RemoteException {
        return mensaje1;
    }

    public String RecibeMensajePc2() throws RemoteException {
        return mensaje2;
    }

    public void EnviaAlias(String alias, int idpc) throws RemoteException {
        if(idpc == 1)
            alias1 = alias;
        if(idpc == 2)
            alias2 = alias;

        System.out.println("Se ha conectado: "+alias);
    }

    public String RecibeAlias1() throws RemoteException {
        return alias1;
    }

    public String RecibeAlias2() throws RemoteException {
        return alias2;
    }
}
