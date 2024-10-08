import java.rmi.RemoteException;
import java.rmi.Remote;

public interface IObjetoRemoto1 extends Remote {
    float suma(float num1, float num2) throws RemoteException;
    void imprime(float valor) throws RemoteException;
    void saludo(String text) throws RemoteException;
}