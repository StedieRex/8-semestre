

import java.rmi.Naming;
import java.util.Scanner;

public class Cliente2 {
    public static void main(String []args) {
        String mensaje1, mensaje2, alias1, alias2, url;
        Scanner lectura = new Scanner(System.in);
        Cifrado cifra = new Cifrado();

        try {
            System.setProperty("Java.security.policy", "rmi.policy");
            url = "rmi://" + args[0] + ":2320/SistemasDistribuidos";
            IObjetoRemoto objrem = (IObjetoRemoto) Naming.lookup(url);

            System.out.println("Chat - Computadora 2");
            System.out.println("Dame el alias: ");
            alias2 = lectura.nextLine();
            objrem.EnviaAlias(alias2, 1);
            do {
                alias1 = objrem.RecibeAlias2();
            } while (alias1 == null);

            do {
                System.out.print(alias2 + ": ");
                mensaje2 = lectura.nextLine();
                mensaje2 = cifra.Cifrar(mensaje2, 3);
                objrem.EnviaMensaje(mensaje2, 1);
                mensaje1 = objrem.RecibeMensajePc2();
                if (mensaje1 != null) {
                    mensaje1 = cifra.Descifrar(mensaje1, 3);
                    System.out.println(alias1 + ": " + mensaje1);
                }
            } while (!mensaje2.isEmpty());
        } catch (Exception e) {
            System.out.println("Se perdió la conexión con el servidor");
        }
    }// Fin main
}// fin cliente 1