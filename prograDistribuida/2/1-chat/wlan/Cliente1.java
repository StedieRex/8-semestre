import java.rmi.Naming;
import java.util.Scanner;

public class Cliente1 {
    public static void main(String[] args) {
        String mensaje1, mensaje2, alias1, alias2, url;

        Scanner lectura = new Scanner(System.in);
        Cifrado cifra = new Cifrado();

        try {
            // Configura la política de seguridad
            System.setProperty("java.security.policy", "rmi.policy");

            // args[0] debe ser la IP del servidor (por ejemplo, "192.168.1.100")
            url = "rmi://" + args[0] + ":2320/SistemasDistribuidos";
            IObjetoRemoto objrem = (IObjetoRemoto) Naming.lookup(url);

            System.out.println("Chat - Computadora 1");
            System.out.println("Dame el alias: ");
            alias1 = lectura.nextLine();

            // Envia alias a la otra computadora
            objrem.EnviaAlias(alias1, 2);

            do {
                alias2 = objrem.RecibeAlias1();
            } while (alias2 == null);

            do {
                System.out.print(alias1 + ": ");
                mensaje1 = lectura.nextLine();
                mensaje1 = cifra.Cifrar(mensaje1, 3);
                objrem.EnviaMensaje(mensaje1, 2);

                mensaje2 = objrem.RecibeMensajePc1();
                if (mensaje2 != null) {
                    mensaje2 = cifra.Descifrar(mensaje2, 3);
                    System.out.println(alias2 + ": " + mensaje2);
                }
            } while (!mensaje1.isEmpty());

        } catch (Exception e) {
            System.out.println("Se perdió la conexión con el servidor");
        }
    }
}
