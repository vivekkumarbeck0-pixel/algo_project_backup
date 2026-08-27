1\. Poore System ka Architecture \& File Map

Aapke project ke andar har ek file ka kya kaam hai aur woh kis-kis se judi hui hai, iski complete list yeh rahi:



trading\_crude.py (The Core Brain / Main Engine)



Kaam: Live market data fetch karna, Angel One API se connect rehna, Technical indicators / Open Interest (OI) / Scenario / Pivot calculate karna, aur Trade Buy/Sell ka decision lena.



Kis se juda hai: config.py, sheets\_logger.py, aur Angel One WebSocket/API se.



sheets\_logger.py (The Data Recorder)



Kaam: Jab bhi koi trade entry ya exit hoti hai, uska poora data (Timestamp, Symbol, Action, Prices, PnL, OI, Scenario, Pivot, Volume) Google Sheets par push karna.



Kis se juda hai: trading\_crude.py aur Google Sheets API (service\_account.json).



config.py \& .env (The Credentials \& Settings)



Kaam: API keys, client codes, passwords, aur Google Sheet URL ko securely store karke rakhna.



Kis se juda hai: Saari python scripts inhi se credentials uthati hain.



service\_account.json (Google Access Key)



Kaam: Google Cloud service account ki private key, jiske zariye python script bina kisi login ke aapki Google Sheet me data write kar pati hai.



2\. Render Server Setup \& Linkages (Hamesha Chalu Rakhne Ka Tarika)

Render par aapka bot hamesha active rahe aur "sone" (spin down) na de, iske liye ye structure kaam karta hai:



Background Worker / Web Service on Render:



Render par aapne project ko deploy kiya hua hai (algo\_project\_backup).



Free Tier Limitation \& Uptime Trick:



Render ke free instances par agar kuch minutes tak koi external HTTP web request na aaye, toh wo sleep mode me chale jate hain.



Isko active rakhne ke liye ya toh aapne background worker process configure kiya hoga, ya phir kisi external Uptime monitor (jaise UptimeRobot) ka use karke har 5-10 minute me server ko ping karwa rahe hain taaki server hamesha awake rahe aur market hours me 24/7 data log hota rahe.



3\. Data Flow: Kahan se Data Kahan Jata hai? (Workflow Link)

Live Market (MCX / Angel One API) → trading\_crude.py (Ticks aur candles read karta hai).



Signal Generation → Jab condition match hoti hai, bot Buy/Sell order place karta hai.



Google Sheets Logging → Trade exit hone par sheets\_logger.py active hota hai, service\_account.json ke through permission leta hai, aur seedha Google Sheet (Crude\_Algo\_Trade\_Logs) me ek poori combined row write kar deta hai.



Aap is structure ko chahe toh apne notes me save karke rakh sakte hain. Isse aapko kabhi bhi future me debug karna ho ya naya bot banana ho, toh turant samajh aa jayega ki kaun si cheez kahan par connected hai!

