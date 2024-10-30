import asyncio
from bleak import BleakClient

# Dirección MAC del ESP32 (reemplaza con la tuya)
esp32_address = "24:AD:17:A3:02:09"

# UUID de la característica definida en el código del ESP32
CHARACTERISTIC_UUID = "2A6E"

async def connect_to_esp32():
    async with BleakClient(esp32_address) as client:
        if await client.is_connected():
            print(f"Conectado a {esp32_address}")

            # Leer el valor de la característica
            value = await client.read_gatt_char(CHARACTERISTIC_UUID)
            print(f"Valor recibido: {value}")

            # Si deseas enviar datos al ESP32, puedes usar:
            # await client.write_gatt_char(CHARACTERISTIC_UUID, bytearray([0x01, 0x02]))

            # Para notificaciones, puedes definir una callback
            def notification_handler(sender, data):
                print(f"Notificación de {sender}: {data}")

            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

            # Mantener la conexión durante un tiempo
            await asyncio.sleep(10)

            # Detener las notificaciones
            await client.stop_notify(CHARACTERISTIC_UUID)

# Ejecutar la función principal
loop = asyncio.get_event_loop()
loop.run_until_complete(connect_to_esp32())
