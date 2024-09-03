import java.rmi.Naming;

public class cliente {
    public static void main(String[] args) {
        float r;
        try{
            System.setProperty("java.security.policy", "rmi.policy");
            System.setSecurityManager(new SecurityManager());

            IObjetetoRemoto1 objrem1 = (IObjetetoRemoto1)Naming.lookup("rmi://localhost:2320/sistemaDistribuido1");

            r = objrem1.suma(10, 20);
            System.out.println("La suma es: "+r);
            objrem1.imprime(100);
            objrem1.saludo("Hola desde el cliente");
        }catch(Exception e){
            e.printStackTrace();
        }
    }    
}
