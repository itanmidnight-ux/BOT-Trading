//+------------------------------------------------------------------+
//|                                          expert_advisor.mq5      |
//|                                                                    |
//| Puente de ejecucion Python <-> MT5 para el bot XAUUSD 1m.          |
//|                                                                    |
//| Este EA es la via ALTERNATIVA de ejecucion (se activa poniendo     |
//| USE_EA_BRIDGE=True en config.py). Por defecto el bot opera con la  |
//| API python `MetaTrader5` directa y este EA no hace falta. Usalo    |
//| cuando tu broker/setup requiere que las ordenes se originen desde  |
//| un EA dentro del terminal.                                         |
//|                                                                    |
//| Protocolo de archivos (carpeta Common\Files\bot_bridge\, ver el    |
//| detalle en mt5_connector/bridge.py):                               |
//|   commands/<id>.cmd   -> Python escribe, este EA lee y borra.      |
//|   acks/<id>.ack        -> este EA escribe el resultado de cada cmd.|
//|   status/status.status -> este EA publica cuenta + posiciones.     |
//|                                                                    |
//| IMPORTANTE: este archivo debe compilarse con MetaEditor (F7) y     |
//| correr en el Strategy Tester / terminal real antes de confiar en   |
//| el, igual que cualquier EA. No fue compilado ni ejecutado en el    |
//| entorno donde se genero este codigo (sin MetaEditor/MT5 disponible |
//| alli); revisa la compilacion y hace forward-test en cuenta demo    |
//| antes de production.                                               |
//+------------------------------------------------------------------+
#property copyright   ""
#property version     "1.00"

#include <Trade\Trade.mqh>

input long InpMagic            = 990099;   // debe coincidir con BRIDGE_MAGIC en config.py
input int  InpTimerMs          = 200;      // cadencia de poll de comandos (ms)
input int  InpStatusEveryTicks = 5;        // cada cuantos timers se re-escribe status.status

CTrade trade;

string CommandsDir = "bot_bridge\\commands";
string AcksDir     = "bot_bridge\\acks";
string StatusDir   = "bot_bridge\\status";

int g_timer_count = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);

   // Usar el filling mode que el simbolo soporta; IOC fijo hace que brokers
   // solo-FOK rechacen todas las ordenes.
   long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) != 0)
      trade.SetTypeFilling(ORDER_FILLING_IOC);
   else if((filling & SYMBOL_FILLING_FOK) != 0)
      trade.SetTypeFilling(ORDER_FILLING_FOK);
   else
      trade.SetTypeFilling(ORDER_FILLING_RETURN);

   // FolderCreate no crea rutas anidadas de una vez: primero el padre.
   FolderCreate("bot_bridge", FILE_COMMON);
   FolderCreate(CommandsDir, FILE_COMMON);
   FolderCreate(AcksDir, FILE_COMMON);
   FolderCreate(StatusDir, FILE_COMMON);

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      Print("ADVERTENCIA: AutoTrading esta deshabilitado en el terminal; el EA no podra ejecutar ordenes.");

   EventSetMillisecondTimer(InpTimerMs);
   Print("expert_advisor.mq5 iniciado. Symbol=", _Symbol, " Magic=", InpMagic);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
}

//+------------------------------------------------------------------+
void OnTick()
{
   // El bridge se procesa por timer (cadencia estable), no por tick:
   // en M1 el flujo de ticks es demasiado irregular para usarlo como reloj.
}

//+------------------------------------------------------------------+
void OnTimer()
{
   ProcessCommands();

   g_timer_count++;
   if(g_timer_count >= InpStatusEveryTicks)
   {
      g_timer_count = 0;
      WriteStatus();
   }
}

//+------------------------------------------------------------------+
//| Escanea commands/ y procesa cada archivo *.cmd                    |
//+------------------------------------------------------------------+
void ProcessCommands()
{
   string filename;
   long handle = FileFindFirst(CommandsDir + "\\*.cmd", filename, FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   do
   {
      ProcessCommandFile(filename);
   }
   while(FileFindNext(handle, filename));

   FileFindClose(handle);
}

//+------------------------------------------------------------------+
void ProcessCommandFile(const string filename)
{
   string path = CommandsDir + "\\" + filename;
   int fh = FileOpen(path, FILE_READ | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      return;

   string type = "", symbol = "", id = "", comment = "";
   long   direction = 0, ticket = 0;
   double volume = 0, sl = 0, tp = 0;

   while(!FileIsEnding(fh))
   {
      string line = FileReadString(fh);
      int eq = StringFind(line, "=");
      if(eq < 0) continue;
      string key = StringSubstr(line, 0, eq);
      string val = StringSubstr(line, eq + 1);

      if(key == "type") type = val;
      else if(key == "symbol") symbol = val;
      else if(key == "direction") direction = StringToInteger(val);
      else if(key == "volume") volume = StringToDouble(val);
      else if(key == "sl") sl = StringToDouble(val);
      else if(key == "tp") tp = StringToDouble(val);
      else if(key == "ticket") ticket = StringToInteger(val);
      else if(key == "id") id = val;
      else if(key == "comment") comment = val;
   }
   FileClose(fh);
   FileDelete(path, FILE_COMMON);

   bool   ok = false;
   string result_comment = "";
   ulong  result_ticket = 0;
   uint   result_retcode = 0;

   if(type == "OPEN")
      ok = ExecuteOpen(symbol, (int)direction, volume, sl, tp, comment, result_ticket, result_retcode, result_comment);
   else if(type == "CLOSE")
      ok = ExecuteClose((ulong)ticket, volume, result_retcode, result_comment);
   else if(type == "MODIFY")
      ok = ExecuteModify((ulong)ticket, sl, tp, result_retcode, result_comment);
   else
      result_comment = "tipo de comando desconocido: " + type;

   WriteAck(id, ok, result_ticket, result_retcode, result_comment);
}

//+------------------------------------------------------------------+
bool ExecuteOpen(const string symbol, const int direction, const double volume, const double sl, const double tp,
                  const string comment, ulong &out_ticket, uint &out_retcode, string &out_comment)
{
   if(symbol != "" && symbol != _Symbol)
   {
      out_comment = "symbol mismatch: esperado " + _Symbol + " recibido " + symbol;
      return false;
   }
   if(volume <= 0)
   {
      out_comment = "volumen invalido";
      return false;
   }

   bool sent;
   if(direction > 0)
      sent = trade.Buy(volume, _Symbol, 0.0, sl, (tp > 0 ? tp : 0.0), comment);
   else
      sent = trade.Sell(volume, _Symbol, 0.0, sl, (tp > 0 ? tp : 0.0), comment);

   out_retcode = trade.ResultRetcode();
   out_ticket  = trade.ResultOrder();
   out_comment = trade.ResultComment();

   return sent && (out_retcode == TRADE_RETCODE_DONE || out_retcode == TRADE_RETCODE_PLACED);
}

//+------------------------------------------------------------------+
bool ExecuteClose(const ulong ticket, const double volume, uint &out_retcode, string &out_comment)
{
   if(!PositionSelectByTicket(ticket))
   {
      out_comment = "posicion no encontrada: " + IntegerToString((long)ticket);
      return false;
   }

   bool sent;
   double pos_volume = PositionGetDouble(POSITION_VOLUME);
   if(volume > 0 && volume < pos_volume)
      sent = trade.PositionClosePartial(ticket, volume);
   else
      sent = trade.PositionClose(ticket);

   out_retcode = trade.ResultRetcode();
   out_comment = trade.ResultComment();
   return sent;
}

//+------------------------------------------------------------------+
bool ExecuteModify(const ulong ticket, const double sl, const double tp, uint &out_retcode, string &out_comment)
{
   if(!PositionSelectByTicket(ticket))
   {
      out_comment = "posicion no encontrada: " + IntegerToString((long)ticket);
      return false;
   }

   double final_sl = (sl >= 0 ? sl : PositionGetDouble(POSITION_SL));
   double final_tp = (tp >= 0 ? tp : PositionGetDouble(POSITION_TP));

   bool sent = trade.PositionModify(ticket, final_sl, final_tp);
   out_retcode = trade.ResultRetcode();
   out_comment = trade.ResultComment();
   return sent;
}

//+------------------------------------------------------------------+
//| Escritura atomica (tmp + move) del ack de un comando               |
//+------------------------------------------------------------------+
void WriteAck(const string id, const bool ok, const ulong ticket, const uint retcode, const string comment)
{
   if(id == "")
      return;

   string tmp_path = AcksDir + "\\" + id + ".tmp";
   string final_path = AcksDir + "\\" + id + ".ack";

   int fh = FileOpen(tmp_path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      return;

   FileWriteString(fh, "ok=" + (ok ? "1" : "0") + "\n");
   FileWriteString(fh, "ticket=" + IntegerToString((long)ticket) + "\n");
   FileWriteString(fh, "retcode=" + IntegerToString((int)retcode) + "\n");
   FileWriteString(fh, "comment=" + comment + "\n");
   FileClose(fh);

   FileDelete(final_path, FILE_COMMON);
   FileMove(tmp_path, FILE_COMMON, final_path, FILE_COMMON | FILE_REWRITE);
}

//+------------------------------------------------------------------+
//| Publica cuenta + posiciones propias (mismo simbolo/magic)          |
//+------------------------------------------------------------------+
void WriteStatus()
{
   string tmp_path = StatusDir + "\\status.tmp";
   string final_path = StatusDir + "\\status.status";

   int fh = FileOpen(tmp_path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      return;

   FileWriteString(fh, "balance=" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + "\n");
   FileWriteString(fh, "equity=" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + "\n");
   FileWriteString(fh, "margin_free=" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2) + "\n");
   FileWriteString(fh, "spread=" + IntegerToString((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) + "\n");
   FileWriteString(fh, "timestamp=" + IntegerToString((long)TimeCurrent()) + "\n");

   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong pticket = PositionGetTicket(i);
      if(pticket == 0)
         continue;
      if(!PositionSelectByTicket(pticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;

      string line = "POS|" + IntegerToString((long)pticket) + ","
                  + IntegerToString((int)PositionGetInteger(POSITION_TYPE)) + ","
                  + DoubleToString(PositionGetDouble(POSITION_VOLUME), 2) + ","
                  + DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), 5) + ","
                  + DoubleToString(PositionGetDouble(POSITION_SL), 5) + ","
                  + DoubleToString(PositionGetDouble(POSITION_TP), 5) + ","
                  + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + "\n";
      FileWriteString(fh, line);
   }

   FileClose(fh);
   FileDelete(final_path, FILE_COMMON);
   FileMove(tmp_path, FILE_COMMON, final_path, FILE_COMMON | FILE_REWRITE);
}
