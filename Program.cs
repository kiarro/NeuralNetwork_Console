using System;
using System.Threading.Tasks;

namespace NeuralNetwork_Console
{
    class Program
    {
        static Interface.Handler handler;
        static void Main(string[] args)
        {
            handler = new Interface.Handler();
            Loop();
        }

        static void Loop()
        {
            int a;
            while (true)
            {
                // await command from console
                Console.Write(">> ");
                Task<string> w = Task.Run(() => Console.ReadLine());
                w.Wait();
                // get command
                string command = w.Result;
                command = command.ToLower();
                // process command
                a = handler.ProcessCommand(command);
                // if exit needed then break loop
                if (a == -1) break;
            }
        }
    }
}
