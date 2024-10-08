import java.rmi.*;
import java.rmi.server.*;

public class objetoRemoto1 extends UnicastRemoteObject implements IObjetoRemoto1 {
    public objetoRemoto1() throws RemoteException {
        super();
    }

    public float suma(float num1, float num2) throws RemoteException {
        System.out.println("La suma {Objeto remoto 1} es"+(num1+num2));
        return num1 + num2;
    }

    public void imprime(float valor) throws RemoteException {
        System.out.println("El valor que mandaste al objeto remoto 1 es: " + valor);
    }

    public void saludo(String text) throws RemoteException {
        System.out.println("Hola desde el objeto Remoto 1 con el mensaje del Cleiente" + text);
    }
    
}
